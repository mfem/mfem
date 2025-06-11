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

#ifndef MFEM_PARTICLE_SET
#define MFEM_PARTICLE_SET

#include "../config/config.hpp"

#include "../mesh/mesh.hpp"
#include "fespace.hpp"

#include <vector>

namespace mfem
{



// Struct-of-Arrays ParticleSet
class SoAParticleSet
{
protected:
   const Ordering::Type ordering;
#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

   std::vector<Vector>
   coords; /// Particle coordinate vectors in each spatial dimension
   std::vector<Vector>
   real_fields; /// Additional scalar fields per particle, as specified in constructor

   BlockVector
   v_coords; /// Vector of all particle coordinates ordered by \ref ordering, referencing \ref coords

   void SyncVCoords();

public:
   SoAParticleSet(int dim, int num_fields=0,
               Ordering::Type ordering_=Ordering::Type::byNODES);

#ifdef MFEM_USE_MPI
   SoAParticleSet(MPI_Comm comm_, int dim, int num_fields=0,
               Ordering::Type ordering_=Ordering::Type::byNODES)
      : SoAParticleSet(dim, num_fields, ordering_) { comm = comm_; };
#endif // MFEM_USE_MPI

   /// Get the ordering of particle coordinates returned by \ref GetAllParticles()
   Ordering::Type GetOrdering() const { return ordering; };

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const  { return coords[0].Size(); };

   /// Get the number of fields per particle for this ParticleSet.
   int GetNF() const { return real_fields.size(); };

   /** Initialize particles randomly within bounding box defined by input \p m . All new scalar fields are set to 0.
       @param[in] m               Mesh defining bounding box to initialize particles on.
       @param[in] num_particles   Number of particles to add to ParticleSet.
       @param[in] reset           (Optional) Delete all existing particles if true, add to existing particles if false.
       @param[in] seed            (Optional) Seed.*/
   void RandomInitialize(Mesh &m, int num_particles, bool reset=false, int seed=0);

   /// Add particle(s) specified by \p in_coords following \ref ordering, with field data given by \p in_fields . Number of fields must match that specified in object construction.
   void AddParticles(const Vector &in_coords, const Vector *in_fields[]= {});

   /// Remove particle(s) specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get particle \p i 's coordinates.
   void GetParticle(int i, Vector &pos) const;

   /// Get reference to the internal particle coordinates Vector.
   const Vector& GetAllParticles() const { return v_coords; };

   /// Get particle \p i , field \p f value.
   const real_t GetParticleField(int i, int f) const { return real_fields[f][i]; };

   /// Get reference to the internal field \p f Vector for all particles.
   const Vector& GetAllParticlesField(int f) const { return real_fields[f]; };

   /// Set particle \p i , field \p f value to \p val.
   void SetParticleField(int i, int f, real_t val) { real_fields[f][i] = val; };

   void SetAllParticlesField(int f, const Vector &field_vals) { real_fields[f] = field_vals; };

   /// Update particle positions using given new coordinates \p new_coords .
   void UpdateParticlePositions(const Vector &new_coords);

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI

};

} // namespace mfem


#endif // MFEM_PARTICLE_SET