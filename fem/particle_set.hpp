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

template<int Dim, int VFields=0, int SFields=0>
struct Particle
{
   real_t coords[Dim];
   real_t vector_fields[VFields][Dim];
   real_t scalar_fields[SFields];
};

template<typename T>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T> requires that T is a Particle."); };

template<int Dim, int VFields, int SFields>
class ParticleSet<Particle<Dim,VFields,SFields>>
{

public:

protected:

#ifdef MFEM_USE_MPI

   MPI_Comm comm;

#endif // MFEM_USE_MPI


#ifdef MFEM_PARTICLES_SOA

   std::array<std::vector<real_t>, Dim> coords; /// Particle coordinate vectors in each spatial dimension
   std::array<std::array<std::vector<real_t>, Dim>, VFields> vector_fields; /// Particle vector fields
   std::array<std::vector<real_t>, SFields> scalar_fields; /// Particle scalar fields

#else 

   std::vector<Particle<Dim, VFields, SFields>> particles;

#endif // MFEM_PARTICLES_SOA

   BlockVector v_coords; /// Vector reference to or copy of all particle coordinates ordered byNODES, for easy input to FindPointsGSLib

   void SyncVCoords();

public:
   /// Optionally reserve memory for \p reserve particles.  
   ParticleSet(int reserve=0);

#ifdef MFEM_USE_MPI
   /// Optionally reserve memory for \p reserve particles on this rank.
   ParticleSet(MPI_Comm comm_, int rank_reserve=0) : ParticleSet(rank_reserve) { comm = comm_; };
#endif // MFEM_USE_MPI

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const;

   /// Add particle
   void AddParticle(const Particle<Dim, VFields, SFields> &p);

   /// Remove particle(s) specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get particle \p i data
   void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const;

   /// Get const reference to the internal particle coordinates Vector.
   const Vector& GetParticleCoords() const { SyncVCoords(); return v_coords; };

   /// Update particle positions using given new coordinates \p new_coords , with ordering byNODES.
   void UpdateParticlePositions(const Vector &new_coords);

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI

};

#ifdef MFEM_PARTICLES_SOA

// Struct-of-Arrays implementation:

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::SyncVCoords()
{
   if (GetNP()*Dim != v_coords.Size()) // Re-allocate if size of v_coords is inconsistent with number of particles
   {
      mfem::Array<int> bOffsets(Dim);
      for (int d = 0; d < Dim; d++)
         bOffsets[d] = d*GetNP();
      v_coords = BlockVector(bOffsets);
      for (int d = 0; d < Dim; d++)
         v_coords.GetBlock(d).SetDataAndSize(coords[d].data(), coords[d].size()); // !! Just set ptr to data !!
   }
}

template<int Dim, int VFields, int SFields>
ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet(int reserve)
{
   // Reserve memory:
   for (int d = 0; d < Dim; d++)
   {
      coords[d].reserve(reserve);
      for (int v = 0; v < VFields; v++)
         vector_fields[v].reserve(reserve);
   }
   for (int s = 0; s < SFields; s++)
      scalar_fields[s].reserve(reserve);
}

template<int Dim, int VFields, int SFields>
int ParticleSet<Particle<Dim, VFields, SFields>>::GetNP() const { return coords[0].size(); }

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::AddParticle(const Particle<Dim, VFields, SFields> &p)
{
   // !! Add data to multiple arrays !!
   for (int d = 0; d < Dim; d++)
   {
      coords[d].push_back(p.coords[d]);
      for (int v = 0; v < VFields; v++)
         vector_fields[v][d].push_back(p.vector_fields[v][d]);
   }
   for (int s = 0; s < SFields; s++)
      scalar_fields[s].push_back(p.scalar_fields[s]);

}

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::RemoveParticles(const Array<int> &list)
{
   int num_old = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   int rm_count = 0;
   for (int i = sorted_list[rm_count]; i < num_old; i++)
   {
      if (rm_count < sorted_list.Size() && i == sorted_list[rm_count])
      {
         rm_count++;
      }
      else
      {
         // Shift elements rm_count
         // !! Multiple arrays shifting !!
         for (int d = 0; d < Dim; d++)
         {
            coords[d][i-rm_count] = coords[d][i];
            for (int v = 0; v < VFields; v++)
               vector_fields[v][d][i-rm_count] = vector_fields[v][d][i];
         }
         for (int s = 0; s < SFields; s++)
            scalar_fields[s][i-rm_count] = scalar_fields[s][i];
      }
   }

   // Resize / remove tails
   // !! Multiple arrays resizing !!
   int num_new = num_old - list.Size();
   for (int d = 0; d < Dim; d++)
   {
      coords[d].resize(num_new);
      for (int v = 0; v < VFields; v++)
         vector_fields[v][d].resize(num_new);
   }
   for (int s = 0; s < SFields; s++)
      scalar_fields[s].resize(num_new);
}


template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::GetParticle(int i, Particle<Dim, VFields, SFields> &p) const
{
   // !! Get data from multiple arrays !!
   for (int d = 0; d < Dim; d++)
   {
      p.coords[d] = coords[d][i];
      for (int v = 0; v < VFields; v++)
         p.vector_fields[v][d] = vector_fields[v][d][i];
   }
   for (int s = 0; s < SFields; s++)
      p.scalar_fields[s] = scalar_fields[s][i];
}

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::UpdateParticlePositions(const Vector &new_coords)
{
   // Copy data over
   // !! (inner loop well-suited for OpenMP ? cache hits on CPU? (if it matters)) !!
   for (int d = 0; d < Dim; d++)
   {
      for (int i = 0; i < GetNP(); i++)
         coords[d][i] = new_coords[i+d*GetNP()];
   }
}


#else

// Array-of-Structs implementation:

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::SyncVCoords()
{
   if (GetNP()*Dim != v_coords.Size()) // Re-allocate if size of v_coords is inconsistent with number of particles
   {
      mfem::Array<int> bOffsets(Dim);
      for (int d = 0; d < Dim; d++)
         bOffsets[d] = d*GetNP();
      v_coords = BlockVector(bOffsets);

      // !! Loop over all particles + copy data over (storing GetNP() references isn't worth it) !!
      for (int d = 0; d < Dim; d++)
      {
         for (int i = 0; i < GetNP(); i++)
            v_coords[i + d*GetNP()] = particles[i].coords[d];
      }
   }
}

template<int Dim, int VFields, int SFields>
ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet(int reserve)
{
   // Reserve memory:
   particles.reserve(reserve);
}

template<int Dim, int VFields, int SFields>
int ParticleSet<Particle<Dim, VFields, SFields>>::GetNP() const { return particles.size(); }

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::AddParticle(const Particle<Dim, VFields, SFields> &p)
{
   // !! Add data to single array !!
   particles.push_back(p);
}

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::RemoveParticles(const Array<int> &list)
{
   int num_old = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   int rm_count = 0;
   for (int i = sorted_list[rm_count]; i < num_old; i++)
   {
      if (rm_count < sorted_list.Size() && i == sorted_list[rm_count])
      {
         rm_count++;
      }
      else
      {
         // Shift elements rm_count
         // !! Single array shifting !!
         particles[i-rm_count] = particles[i];
      }
   }

   // Resize / remove tail
   // !! Single array resizing !!
   int num_new = num_old - list.Size();
   particles.resize(num_new);
}


template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::GetParticle(int i, Particle<Dim, VFields, SFields> &p) const
{
   // !! Get data from single array !!
   Particle<Dim, VFields, SFields> p_copy = particles[i];
   p = std::move(p_copy);
}

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim, VFields, SFields>>::UpdateParticlePositions(const Vector &new_coords)
{

   for (int d = 0; d < Dim; d++)
   {
      for (int i = 0; i < GetNP(); i++)
      {
         particles[i].coords[d] = new_coords[i+d*GetNP()]; // !! Consider memory? (better maybe if new_coords byVDIM?) !!
      }
   }
}


#endif // MFEM_PARTICLES_SOA


} // namespace mfem


#endif // MFEM_PARTICLE_SET