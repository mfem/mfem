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
template<int SpaceDim, int NumScalars, int... VectorVDims>
struct Particle
{
public:
   real_t coords[SpaceDim];
   real_t scalars[NumScalars];
   real_t vectors_flat[(VectorVDims + ...)]; // flattened vector data
}

// -----------------------------------------------------------------------------------------------------
// Define Base ParticleSet to restrict first type T
template<typename T, Ordering::Type VOrdering=Ordering::byNODES>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T,VOrdering> requires that T is a Particle."); };


// Define ParticleSet
template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
class ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>
{
public:

   static constexpr Ordering::Type GetOrdering() const { return VOrdering; }

protected:
   static constexpr int TotalComps = SpaceDim + NumScalars + (VectorVDims + ...); // includes comps of coordinates + vectors
   static constexpr int TotalFields = 1 + NumScalars + sizeof...(VectorVDims);
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... };

   std::vector<real_t> data; // Stores ALL particle data
   std::array<Vector, TotalFields> fields; // User-facing Vectors, referencing data

   void SyncVectors();

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

public:
   ParticleSet() = default;

#ifdef MFEM_USE_MPI
   explicit ParticleSet(MPI_Comm comm_) : comm(comm_) {};
#endif // MFEM_USE_MPI

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { data.reserve(res*TotalComps); }

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const { return data.size()/TotalComps; }

   /// Add particle
   void AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p);

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get copy of particle \p i data
   void GetParticle(int i, Particle<SpaceDim, NumScalars, VectorVDims...> &p) const;

   Vector& GetCoords() { return fields[0]; }

   template<int S>
   Vector& GetScalars() { return fields[1+S]; };

   template<int V>
   Vector& GetVectors() { return fields[1+NumScalars+V]; };

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI

};


// -----------------------------------------------------------------------------------------------------
// Fxn implementations:

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::SyncVectors()
{
   // Reset Vector references to data
   int offset = 0;
   int size;
   for (int f = 0; f < TotalFields; f++)
   {
      if (f == 0)
      {
         size = GetNP()*SpaceDim;
      }
      else if (f < 1 + NumScalars)
      {
         size = GetNP();
      }
      else
      {
         size = GetNP()*VDims[f - 1 - NumScalars];
      }

      vfields[f] = Vector(data.data() + offset, size);
      offset += vfields[f].Size();
   }
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p)
{
   int old_np = GetNP();

   if constexpr (VOrdering == Ordering::byNODES)
   {
      real_t *dat;
      for (int c = 0; c < TotalComps; c++)
      {
         if (c < SpaceDim) // If processing coord components
         {
            dat = &p.coords[c];
         }
         else if (c < SpaceDim + NumScalars) // Else if processing scalars
         {
            dat = &p.scalars[c - SpaceDim];
         }
         else // Else processing vector components
         {
            dat = &p.vectors_flat[c - SpaceDim - NumScalars];
         }
         data.insert(data.begin() + old_np*(c+1) + c, *dat);
      }

   }
   else // byVDIM
   {
      // TODO
   }

   SyncVectors();
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::RemoveParticles(const Array<int> &list)
{
   int old_np = GetNP();

   // Sort the indices
   Array<int> sorted_list(list);
   sorted_list.Sort();

   if constexpr (VOrdering == Ordering::byNODES)
   {
      int rm_count = 0;
      for (int i = sorted_list[rm_count]; i < data.size(); i++)
      {
         if (rm_count < sorted_list.Size() && i % GetNP() == sorted_list[rm_count])
         {
            rm_count++;
         }
         else
         {
            data[i-rm_count] = data[i]; // Shift elements rm_count
         }
      }

   }
   else // byVDIM
   {
      // TODO
   }

   // Resize / remove tail
   int num_new = old_np - list.Size();
   data.resize(num_new*TotalComps);

   SyncVectors();
}


template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::GetParticle(int i, Particle<SpaceDim, NumScalars, VectorVDims...> &p) const
{

   if constexpr(VOrdering == Ordering::byNODES)
   {
      real_t *dat;
      for (int c = 0; c < TotalComps; c++)
      {
         if (c < SpaceDim) // If setting coord components
         {
            dat = &p.coords[c];
         }
         else if (c < SpaceDim + NumScalars) // Else if setting scalars
         {
            dat = &p.scalars[c - SpaceDim];
         }
         else // Else setting vector components
         {
            dat = &p.vectors_flat[c - SpaceDim - NumScalars];
         }

         *dat = data[i + c*GetNP()];
      }
   }
   else // byVDIM
   {
      // TODO
   }
}


} // namespace mfem


#endif // MFEM_PARTICLE_SET