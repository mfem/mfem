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

// -----------------------------------------------------------------------------------------------------
// Define Particle struct
template<int Dim, int VFields=0, int SFields=0>
struct Particle
{
   real_t coords[Dim];
   real_t vector_fields[VFields][Dim];
   real_t scalar_fields[SFields];
};

// -----------------------------------------------------------------------------------------------------
// Define Base ParticleSet
template<typename T, Ordering::Type VFieldOrdering>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T,...> requires that T is a Particle."); };

template<int Dim, int VFields, int SFields, Ordering::Type VFieldOrdering>
class ParticleSet<Particle<Dim,VFields,SFields>, VFieldOrdering>
{

protected:

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

   BlockVector v_coords; /// Vector reference to all particle coordinates ordered \ref VFieldOrdering, for easy input to FindPointsGSLib
   std::array<BlockVector, VFields> v_vector_fields; /// Vector reference to all vector fields ordered \ref VFieldOrdering, for user usage

   virtual void SyncVCoords() = 0;
   virtual void SyncVVectorField(int v) = 0;
public:
   ParticleSet() = default;

#ifdef MFEM_USE_MPI
   explicit ParticleSet(MPI_Comm comm_) : comm(comm_) {};
#endif // MFEM_USE_MPI

   static constexpr Ordering::Type GetOrdering() const { return VFieldOrdering; };

   virtual void Reserve(int res) = 0;

   /// Get the number of particles currently held by this ParticleSet.
   virtual int GetNP() const = 0;

   /// Add particle
   virtual void AddParticle(const Particle<Dim, VFields, SFields> &p) = 0;

   /// Remove particle(s) specified by \p list of particle indices.
   virtual void RemoveParticles(const Array<int> &list) = 0;

   /// Get particle \p i data
   virtual void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const = 0;

   /// Get const reference to the internal particle coordinates Vector.
   const Vector& GetParticleCoords() { SyncVCoords(); return v_coords; }; // TODO: make this const ...

   /// Get const reference to the internal vector field \p v Vector
   const Vector& GetVectorField(int v) { SyncVVectorField(v); return v_vector_fields[v]; }; // TODO: make this const ...

   /// Update particle positions using given new coordinates \p new_coords , with ordering \ref VFieldOrdering.
   virtual void UpdateParticlePositions(const Vector &new_coords) = 0;

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI


   virtual ~ParticleSet() = default;

};

// -----------------------------------------------------------------------------------------------------
/// Define Struct-of-Arrays ParticleSet

template<typename T>
class SoAParticleSet : public ParticleSet<T, Ordering::byNODES> {};

template<int Dim, int VFields, int SFields>
class SoAParticleSet<Particle<Dim, VFields, SFields>> : public
                                                      ParticleSet<Particle<Dim,VFields,SFields>, Ordering::byVDIM>
{
protected:

   std::array<std::vector<real_t>, Dim> coords; /// Particle coordinate vectors in each spatial dimension
   std::array<std::array<std::vector<real_t>, Dim>, VFields> vector_fields; /// Particle vector fields
   std::array<std::vector<real_t>, SFields> scalar_fields; /// Particle scalar fields


   void SyncBlockVector(const std::array<std::vector<real_t>, Dim> &actual, BlockVector &bv);
   void SyncVCoords() override;
   void SyncVVectorField(int v) override;

public:
   using ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet;
   
   void Reserve(int res) override;

   int GetNP() const override;

   void AddParticle(const Particle<Dim, VFields, SFields> &p) override;

   void RemoveParticles(const Array<int> &list) override;

   void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const override;

   void UpdateParticlePositions(const Vector &new_coords) override;
};

// -----------------------------------------------------------------------------------------------------
/// Define Array-of-Structs ParticleSet

template<typename T>
class AoSParticleSet : public ParticleSet<T> {};

template<int Dim, int VFields, int SFields>
class AoSParticleSet<Particle<Dim,VFields,SFields>> : public
                                                    ParticleSet<Particle<Dim, VFields, SFields>, Ordering::byVDIM>
{
protected:

   std::vector<Particle<Dim, VFields, SFields>> particles;

   void SyncVCoords() override;
   void SyncVVectorField(int v) override;

public:
   using ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet;

   void Reserve(int res) override;

   int GetNP() const override;

   void AddParticle(const Particle<Dim, VFields, SFields> &p) override;

   void RemoveParticles(const Array<int> &list) override;

   void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const override;

   void UpdateParticlePositions(const Vector &new_coords) override;
};

// -----------------------------------------------------------------------------------------------------
// Struct-of-Arrays implementation:
template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::SyncBlockVector(const std::array<std::vector<real_t>, Dim> &actual, BlockVector &bv)
{
   if (GetNP()*Dim != bv.Size()) // Need to sync if size of bv is inconsistent with number of particles
   {
      mfem::Array<int> bOffsets(Dim);
      for (int d = 0; d < Dim; d++)
      {
         bOffsets[d] = d*GetNP();
      }
      bv = BlockVector(bOffsets);
      for (int d = 0; d < Dim; d++)
      {
         bv.GetBlock(d).SetDataAndSize(const_cast<real_t*>(actual[d].data()), actual[d].size());   // !! Just set ptr to data !!
      }
      // ^^ const cast because block vectors in class should NOT change actual
   }
}

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::SyncVCoords() { SyncBlockVector(coords, this->v_coords); }

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::SyncVVectorField(int v) { SyncBlockVector(vector_fields[v], this->v_vector_fields[v]); }

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::Reserve(int res)
{
   // Reserve memory:
   for (int d = 0; d < Dim; d++)
   {
      coords[d].reserve(res);
      for (int v = 0; v < VFields; v++)
      {
         vector_fields[v][d].reserve(res);
      }
   }
   for (int s = 0; s < SFields; s++)
   {
      scalar_fields[s].reserve(res);
   }

}

template<int Dim, int VFields, int SFields>
int SoAParticleSet<Particle<Dim, VFields, SFields>>::GetNP() const { return coords[0].size(); }

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::AddParticle(const Particle<Dim, VFields, SFields> &p)
{
   // !! Add data to multiple arrays !!
   for (int d = 0; d < Dim; d++)
   {
      coords[d].push_back(p.coords[d]);
      for (int v = 0; v < VFields; v++)
      {
         vector_fields[v][d].push_back(p.vector_fields[v][d]);
      }
   }
   for (int s = 0; s < SFields; s++)
   {
      scalar_fields[s].push_back(p.scalar_fields[s]);
   }

}

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::RemoveParticles(const Array<int> &list)
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
            {
               vector_fields[v][d][i-rm_count] = vector_fields[v][d][i];
            }
         }
         for (int s = 0; s < SFields; s++)
         {
            scalar_fields[s][i-rm_count] = scalar_fields[s][i];
         }
      }
   }

   // Resize / remove tails
   // !! Multiple arrays resizing !!
   int num_new = num_old - list.Size();
   for (int d = 0; d < Dim; d++)
   {
      coords[d].resize(num_new);
      for (int v = 0; v < VFields; v++)
      {
         vector_fields[v][d].resize(num_new);
      }
   }
   for (int s = 0; s < SFields; s++)
   {
      scalar_fields[s].resize(num_new);
   }
}


template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::GetParticle(int i,
                                                                  Particle<Dim, VFields, SFields> &p) const
{
   // !! Get data from multiple arrays !!
   for (int d = 0; d < Dim; d++)
   {
      p.coords[d] = coords[d][i];
      for (int v = 0; v < VFields; v++)
      {
         p.vector_fields[v][d] = vector_fields[v][d][i];
      }
   }
   for (int s = 0; s < SFields; s++)
   {
      p.scalar_fields[s] = scalar_fields[s][i];
   }
}

template<int Dim, int VFields, int SFields>
void SoAParticleSet<Particle<Dim, VFields, SFields>>::UpdateParticlePositions(
                                                     const Vector &new_coords)
{
   // Copy data over
   // !! (inner loop well-suited for OpenMP ? cache hits on CPU? (if it matters)) !!
   for (int d = 0; d < Dim; d++)
   {
      for (int i = 0; i < GetNP(); i++)
      {
         coords[d][i] = new_coords[i+d*GetNP()];
      }
   }
}


// -----------------------------------------------------------------------------------------------------
// Array-of-Structs implementation:

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::SyncVCoords()
{
   if (GetNP()*Dim != this->v_coords.Size()) // Re-allocate if size of v_coords is inconsistent with number of particles
   {
      mfem::Array<int> bOffsets(GetNP());
      for (int i = 0; i < GetNP(); i++)
      {
         bOffsets[i] = i*Dim;
      }
      this->v_coords = BlockVector(bOffsets);

      // !! Loop over all particles + store references to each particles' coords !!
      for (int i = 0; i < GetNP(); i++)
      {
         this->v_coords.GetBlock(i).SetDataAndSize(particles[i].coords, Dim);
      }
   }
}

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::SyncVVectorField(int v)
{
   if (GetNP()*Dim != this->v_vector_fields[v].Size()) // Re-allocate if size of v_vector_fields is inconsistent with number of particles
   {
      mfem::Array<int> bOffsets(GetNP());
      for (int i = 0; i < GetNP(); i++)
      {
         bOffsets[i] = i*Dim;
      }
      this->v_vector_fields = BlockVector(bOffsets);

      // !! Loop over all particles + store references to each particles' coords !!
      for (int i = 0; i < GetNP(); i++)
      {
         this->v_vector_fields.GetBlock(i).SetDataAndSize(particles[i].coords, Dim);
      }
   }
}

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::Reserve(int res)
{  
   particles.reserve(res);
}

template<int Dim, int VFields, int SFields>
int AoSParticleSet<Particle<Dim, VFields, SFields>>::GetNP() const { return particles.size(); }

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::AddParticle(const Particle<Dim, VFields, SFields> &p)
{
   // !! Add data to single array !!
   particles.push_back(p);
}

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::RemoveParticles(const Array<int> &list)
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
void AoSParticleSet<Particle<Dim, VFields, SFields>>::GetParticle(int i, Particle<Dim, VFields, SFields> &p) const
{
   // !! Get data from single array !!
   Particle<Dim, VFields, SFields> p_copy = particles[i];
   p = std::move(p_copy);
}

template<int Dim, int VFields, int SFields>
void AoSParticleSet<Particle<Dim, VFields, SFields>>::UpdateParticlePositions(const Vector &new_coords)
{
   for (int i = 0; i < GetNP(); i++)
   {
      for (int d = 0; d < Dim; d++)
      {
         particles[i].coords[d] = new_coords[d+i*GetNP()];
      }
   }
}

} // namespace mfem


#endif // MFEM_PARTICLE_SET