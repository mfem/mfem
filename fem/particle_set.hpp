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
#include <numeric>

namespace mfem
{

// -----------------------------------------------------------------------------------------------------
// Define Particle class

template<int SpaceDim, int NumScalars, int... VectorVDims>
class Particle
{
private:
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... };

   const bool owning;

   Vector coords;
   std::array<real_t*, NumScalars> scalars;
   std::array<Vector, sizeof...(VectorVDims)> vectors;

public:
   /// Create a new particle which owns its own data
   Particle()
   : owning(true)
   {
      coords.SetSize(SpaceDim);

      // Initialize scalar ptrs
      for (int i = 0; i < scalars.size(); i++)
         scalars[i] = new real_t(0.0);

      // Initialize vectors
      for (int i = 0; i < vectors.size(); i++)
         vectors[i].SetSize(VDims[i]);
   }

   // Create a new particle whose data references external data
   Particle(Vector *in_coords, real_t *in_scalars[], Vector *in_vectors[])
   : owning(false)
   {
      coords = Vector(in_coords->GetData(), SpaceDim);
      for (int i = 0; i < scalars.size(); i++)
         scalars[i] = &in_scalars[i];
      
      for (int i = 0; i < vectors.size(); i++)
         vectors[i] = Vector(in_vectors[i]->GetData(), VDims[i]);
   }

   Vector& Coords() { return coords; };

   real_t& Scalar(int s) { return scalars[s]; };

   Vector& Vector(int v) { return vectors[v]; };

   const Vector& Coords() const { return coords; };

   const real_t& Scalar(int s) const { return scalars[s]; };

   const Vector& Vector(int v) const { return vectors[v]; };


   ~Particle()
   {
      if (owning)
      {
         for (int i = 0; i < NumScalars; i++)
            delete scalars[i];
      }
   }
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

   static constexpr 
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

   /// Get particle \p i . **If \ref VOrdering is byNODES, Particle holds copy of data. If \ref VOrdering is byVDIM, Particle references data.**
   Particle<SpaceDim, NumScalars, VectorVDims...> GetParticle(int i) const;

   /// Set particle \p i 's data.
   void SetParticle(int i, const Particle<SpaceDim, NumScalars, VectorVDims...> &p);

   Vector& GetSetCoords() { return fields[0]; }

   Vector& GetSetScalars(int s) { return fields[1+s]; };

   Vector& GetSetVectors(int v) { return fields[1+NumScalars+v]; };

   const Vector& GetSetCoords() const { return fields[0]; };

   const Vector& GetSetScalars(int s) const { return fields[1+s]; };

   const Vector& GetSetVectors(int v) const { return fields[1+NumScalars+v]; };

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
      real_t dat;
      int offset = old_np;

      std::array<int, sizeof...(VectorVDims)> scannedVDims;
      std::inclusive_scan(VDims.begin(), VDims.end(), scannedVDims.begin());
      int cur_vec = 0;

      for (int c = 0; c < TotalComps; c++)
      {
         if (c < SpaceDim) // If processing coord components
         {
            dat = p.Coords()[c];
         }
         else if (c < SpaceDim + NumScalars) // Else if processing scalars
         {
            dat = p.Scalar(c - SpaceDim);
         }
         else // Else processing vector components
         {
            int vc = c - SpaceDim - NumScalars;
            if (vc == scannedVDims[cur_vec])
               cur_vec++;
            int comp = scannedVDims[cur_vec] - vc;
            dat = p.Vector(cur_vec)[comp];
         }
         data.insert(data.begin() + offset, dat);
         offset += old_np + 1; // 1 to account for added data
      }

   }
   else // byVDIM
   {
      std::array<int, sizeof...(VectorVDims)> scannedVDims;
      std::exclusive_scan(VDims.begin(), VDims.end(), scannedVDims.begin());

      // Insert coords
      real_t *coords = p.Coords().GetData();
      data.insert(data.begin() + old_np*SpaceDim, coords, coords + SpaceDim);

      // Insert scalars
      for (int s = 0; s < NumScalars; s++)
      {
         data.insert(data.begin() + (old_np+1)*SpaceDim + old_np*(s+1) + s, p.Scalar(s));
      }

      // Insert vectors
      for (int v = 0; v < sizeof...(VectorVDims); v++)
      {
         real_t *vec = p.Vector(v).GetData();
         data.insert(data.begin() + (old_np+1)*SpaceDim*NumScalars + old_np*(scannedVDims[v] + VDims[v]) + scannedVDims[v], vec, vec + VDims[v]);
      }
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
      for (int f = 0; c < TotalComps; c++)
      {
         if (c < SpaceDim) // If setting coord components
         {
            dat = &p.Coords()[c];
         }
         else if (c < SpaceDim + NumScalars) // Else if setting scalars
         {
            dat = &p.Scalar(c - SpaceDim);
         }
         else // Else setting vector components
         {

            dat = &p.Vector(c - SpaceDim - NumScalars];
         }

         *dat = data[i + c*GetNP()];
      }
   }
   else // byVDIM
   {
      // TODO
   }
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::SetParticle(int i, const Particle<SpaceDim, NumScalars, VectorVDims...> &p) const
{

   if constexpr(VOrdering == Ordering::byNODES)
   {
      const real_t *dat;
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

         data[i + c*GetNP()] = *dat;
      }
   }
   else // byVDIM
   {
      // TODO
   }
}

} // namespace mfem


#endif // MFEM_PARTICLE_SET