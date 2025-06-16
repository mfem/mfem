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
template<typename T>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T> requires that T is a Particle."); };

template<int Dim, int VFields, int SFields>
class ParticleSet<Particle<Dim,VFields,SFields>>
{

protected:

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI


   void UpdateFromVVector(const Vector &new_field, std::function<real_t&(int,int)> getRef, Ordering::Type ordering);

public:
   ParticleSet() = default;

#ifdef MFEM_USE_MPI
   explicit ParticleSet(MPI_Comm comm_) : comm(comm_) {};
#endif // MFEM_USE_MPI

   virtual void Reserve(int res) = 0;

   /// Get the number of particles currently held by this ParticleSet.
   virtual int GetNP() const = 0;

   /// Add particle
   virtual void AddParticle(const Particle<Dim, VFields, SFields> &p) = 0;

   /// Remove particle(s) specified by \p list of particle indices.
   virtual void RemoveParticles(const Array<int> &list) = 0;

   /// Get copy of particle \p i data
   virtual void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const = 0;

   /// Get reference to particle \p i 's position component \p comp
   virtual real_t& GetCoord(int i, int comp) = 0;

   /// Get reference to particle \p i 's \p v vector component \p comp
   virtual real_t& GetVector(int v, int i, int comp) = 0;

   /// Get reference to particle \p i 's \p S scalar
   virtual real_t& GetScalar(int s, int i) = 0;

   /// Update all particle coordinates
   void UpdateCoords(const Vector &new_coords, Ordering::Type ordering)
   {
      UpdateFromVVector(new_coords,
         [&](int i, int comp) -> real_t& { return GetCoord(i, comp); },
         ordering);
   };

   /// Update all particles' \p v vector field
   void UpdateVF(int v, const Vector &new_vf, Ordering::Type ordering)
   {
      UpdateFromVVector(new_vf, 
         [&](int i, int comp) -> real_t& { return GetVector(v, i, comp); }, 
         ordering);
   };

   /// Update all particles' \p s scalar field
   void UpdateSF(int s, const Vector &new_sf)
   {
      for (int i = 0; i < GetNP(); i++)
         GetScalar(s,i) = new_sf[i];
   }

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI


   virtual ~ParticleSet() = default;

};

template<int Dim, int VFields, int SFields>
void ParticleSet<Particle<Dim,VFields,SFields>>::UpdateFromVVector(const Vector &new_field, std::function<real_t&(int,int)> getRef, Ordering::Type ordering)
{
   // Copy data over
   if (ordering == Ordering::byNODES)
   {
      for (int d = 0; d < Dim; d++)
      {
         for (int i = 0; i < GetNP(); i++)
         {
            getRef(i,d) = new_field[i+d*GetNP()];
         }
      }
   }
   else // byVDIM
   {
      for (int i = 0; i < GetNP(); i++)
      {
         for (int d = 0; d < Dim; d++)
         {
            getRef(i,d) = new_field[d+i*Dim];
         }
      }
   }
}




// -----------------------------------------------------------------------------------------------------
/// Define Struct-of-Arrays ParticleSet

template<typename T>
class SoAParticleSet : public ParticleSet<T> {};

template<int Dim, int VFields, int SFields>
class SoAParticleSet<Particle<Dim, VFields, SFields>> : public
                                                      ParticleSet<Particle<Dim,VFields,SFields>>
{
protected:

   std::array<std::vector<real_t>, Dim> coords; /// Particle coordinate vectors in each spatial dimension
   std::array<std::array<std::vector<real_t>, Dim>, VFields> vector_fields; /// Particle vector fields
   std::array<std::vector<real_t>, SFields> scalar_fields; /// Particle scalar fields
   
public:
   using ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet;
   
   void Reserve(int res) override;

   int GetNP() const override { coords[0].size(); }

   void AddParticle(const Particle<Dim, VFields, SFields> &p) override;

   void RemoveParticles(const Array<int> &list) override;

   void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const override;

   real_t& GetCoord(int i, int comp) override { return coords[comp][i]; }

   real_t& GetVector(int v, int i, int comp) override { return vector_fields[v][comp][i]; }

   real_t& GetScalar(int s, int i) override { return scalar_fields[s][i]; }



   // Bonus functions we can leverage with SoA:

   const real_t* GetParticleCoords(int comp) const { return coords[comp].data(); }

   const real_t* GetVectorField(int v, int comp) const { return vector_fields[v][comp].data(); }

   const real_t* GetScalarField(int s) const { return scalar_fields[s].data(); }

   real_t* GetParticleCoords(int comp) { return coords[comp].data(); }

   real_t* GetVectorField(int v, int comp) { return vector_fields[v][comp].data(); }

   real_t* GetScalarField(int s) { return scalar_fields[s].data(); }

};

// -----------------------------------------------------------------------------------------------------
/// Define Array-of-Structs ParticleSet

template<typename T>
class AoSParticleSet : public ParticleSet<T> {};

template<int Dim, int VFields, int SFields>
class AoSParticleSet<Particle<Dim,VFields,SFields>> : public
                                                    ParticleSet<Particle<Dim, VFields, SFields>>
{
protected:

   std::vector<Particle<Dim, VFields, SFields>> particles;

public:
   using ParticleSet<Particle<Dim, VFields, SFields>>::ParticleSet;

   void Reserve(int res) override { particles.reserve(res); }

   int GetNP() const override { return particles.size(); }

   void AddParticle(const Particle<Dim, VFields, SFields> &p) override { particles.push_back(p); }

   void RemoveParticles(const Array<int> &list) override;

   void GetParticle(int i, Particle<Dim, VFields, SFields> &p) const override;

   real_t& GetCoord(int i, int comp) override { return particles[i].coords[comp]; }

   real_t& GetVector(int v, int i, int comp) override { return particles[i].vector_fields[v][comp]; }

   real_t& GetScalar(int s, int i) override { return particles[i].scalar_fields[s]; }



   // Bonus functions we can leverage with AoS:

   const Particle<Dim, VFields, SFields>* GetParticles() const { return particles.data(); };

   Particle<Dim, VFields, SFields>* GetParticles() { return particles.data(); };
};

// -----------------------------------------------------------------------------------------------------
// Struct-of-Arrays implementation:

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


// -----------------------------------------------------------------------------------------------------
// Array-of-Structs implementation:

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

} // namespace mfem


#endif // MFEM_PARTICLE_SET