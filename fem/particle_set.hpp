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

template<int SpaceDim, int NumScalars=0, int... VectorVDims>
class Particle
{
private:
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... };

   const bool owning;

   Vector coords;
   std::array<real_t*, NumScalars> scalars;
   std::array<Vector, sizeof...(VectorVDims)> vectors;

public:
   // static constexpr int SpaceDim() { return SpaceDim; };
   static constexpr int GetNumScalars() { return NumScalars; };
   static constexpr int GetNumVectors() { return sizeof...(VectorVDims); };
   // static constexpr int VDim(int v) { return VDims[v]; };

   /// Create a new particle which owns its own data
   Particle()
   : owning(true)
   {
      coords.SetSize(SpaceDim);
      coords = 0.0;

      // Initialize scalar ptrs
      for (int i = 0; i < scalars.size(); i++)
         scalars[i] = new real_t(0.0);

      // Initialize vectors
      for (int i = 0; i < vectors.size(); i++)
      {
         vectors[i].SetSize(VDims[i]);
         vectors[i] = 0.0;
      }
   }

   // Create a new particle whose data references external data
   Particle(real_t *in_coords, real_t *in_scalars[], real_t *in_vectors[])
   : owning(false)
   {
      coords = Vector(in_coords, SpaceDim);
      for (int i = 0; i < scalars.size(); i++)
         scalars[i] = in_scalars[i];
      
      for (int i = 0; i < vectors.size(); i++)
         vectors[i] = Vector(in_vectors[i], VDims[i]);
   }

   // Copy constructor
   explicit Particle(const Particle &p)
   : owning(true)
   {
      coords = Vector(p.GetCoords());
      for (int s = 0; s < scalars.size(); s++)
         scalars[s] = new real_t(p.GetScalar(s));
      for (int v = 0; v < vectors.size(); v++)
         vectors[v] = Vector(p.GetVector(v));
   }

   Vector& GetCoords() { return coords; }

   real_t& GetScalar(int s) { return *scalars[s]; }

   Vector& GetVector(int v) { return vectors[v]; }

   const Vector& GetCoords() const { return coords; }

   const real_t& GetScalar(int s) const { return *scalars[s]; }

   const Vector& GetVector(int v) const { return vectors[v]; }

   bool operator==(const Particle &rhs) const
   {
      bool equal = true;
      for (int d = 0; d < SpaceDim; d++)
      {
         if (GetCoords()[d] != rhs.GetCoords()[d])
            equal = false;
      }
      for (int s = 0; s < NumScalars; s++)
      {
         if (GetScalar(s) != rhs.GetScalar(s))
            equal = false;
      }
      for (int v = 0; v < sizeof...(VectorVDims); v++)
      {
         for (int c = 0; c < VDims[v]; c++)
         {
            if (GetVector(v)[c] != rhs.GetVector(v)[c])
               equal = false;
         }
      }
      return equal;
   }

   bool operator!=(const Particle &rhs) const { return !operator==(rhs); };

   ~Particle()
   {
      if (owning)
      {
         for (int i = 0; i < NumScalars; i++)
            delete scalars[i];
      }
   }

   void Print(std::ostream &out=mfem::out)
   {
      out << "Coords: (";
      for (int d = 0; d < SpaceDim; d++)
         out << GetCoords()[d] << ( (d+1 < SpaceDim) ? "," : ")\n");
      for (int s = 0; s < NumScalars; s++)
         out << "Scalar " << s << ": " << GetScalar(s) << "\n";
      for (int v = 0; v < sizeof...(VectorVDims); v++)
      {
         out << "Vector " << v << ": (";
         for (int c = 0; c < VDims[v]; c++)
            out << GetVector(v)[c] << ( (c+1 < VDims[v]) ? "," : ")\n");
      }
   }
};

// -----------------------------------------------------------------------------------------------------
// Define Base ParticleSet to restrict first type T
template<typename T, Ordering::Type VOrdering=Ordering::byNODES>
class ParticleSet { static_assert(sizeof(T)==0, "ParticleSet<T,VOrdering> requires that T is a Particle."); };


// Define ParticleSet
template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
class ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>
{
public:

   static constexpr Ordering::Type GetOrdering() { return VOrdering; }

protected:
   static constexpr std::array<int, sizeof...(VectorVDims)> VDims = { VectorVDims... }; // VDims of vectors
   static constexpr int TotalFields = 1 + NumScalars + sizeof...(VectorVDims); // Total number of fields / particle
   static constexpr int TotalComps = SpaceDim + NumScalars + (VectorVDims + ... + 0); // Total comps / particle
   
   static constexpr std::array<int, TotalFields> MakeFieldVDims()
   {
      std::array<int, TotalFields> temp{};
      temp[0] = SpaceDim;
      for (int s = 0; s < NumScalars; s++) temp[1+s] = 1;
      for (int v = 0; v < sizeof...(VectorVDims); v++) temp[1+NumScalars+v] = ParticleSet::VDims[v];
      return temp;
   }
   static constexpr std::array<int, TotalFields> FieldVDims = MakeFieldVDims(); // VDims of all fields (coords, scalars, vectors...)
   
   static const std::array<int, TotalFields> MakeExclScanFieldVDims() // std::exclusive_scan not constexpr until C++20
   {
      std::array<int, TotalFields> temp{};
      std::exclusive_scan(FieldVDims.begin(), FieldVDims.end(), temp.begin(), 0);
      return temp;
   }
   static inline const std::array<int, TotalFields> ExclScanFieldVDims = MakeExclScanFieldVDims();
   
   std::vector<real_t> data; // Stores ALL particle data
   std::array<Vector, TotalFields> fields; // User-facing Vectors, referencing data

   void SyncVectors();

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif // MFEM_USE_MPI

public:

   ParticleSet() = default;

#ifdef MFEM_USE_MPI
   explicit ParticleSet(MPI_Comm comm_) : comm(comm_) {}
#endif // MFEM_USE_MPI

   /// Reserve room for \p res particles. Can help to avoid re-allocation for adding + removing particles.
   void Reserve(int res) { data.reserve(res*TotalComps); }

   /// Get the number of particles currently held by this ParticleSet.
   int GetNP() const { return data.size()/TotalComps; }

   /// Add particle
   void AddParticle(const Particle<SpaceDim, NumScalars, VectorVDims...> &p);

   /// Remove particle data specified by \p list of particle indices.
   void RemoveParticles(const Array<int> &list);

   /// Get particle \p i referencing actual, data ONLY FOR Ordering byVDIM.
   Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleRef(int i);

   const Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleRef(int i) const { return const_cast<ParticleSet*>(this)->GetParticleRef(i); };
   /// Get copy of data in particle \p i .
   Particle<SpaceDim, NumScalars, VectorVDims...> GetParticleData(int i) const;

   /// Set particle \p i 's data.
   void SetParticle(int i, const Particle<SpaceDim, NumScalars, VectorVDims...> &p);

   Vector& GetSetCoords() { return fields[0]; }

   Vector& GetSetScalars(int s) { return fields[1+s]; }

   Vector& GetSetVectors(int v) { return fields[1+NumScalars+v]; }

   const Vector& GetSetCoords() const { return fields[0]; }

   const Vector& GetSetScalars(int s) const { return fields[1+s]; }

   const Vector& GetSetVectors(int v) const { return fields[1+NumScalars+v]; }

#ifdef MFEM_USE_MPI
   /// Redistribute particles onto ranks specified in \p rank_list .
   void Redistribute(const Array<int> &rank_list);
#endif // MFEM_USE_MPI

   void Print(std::ostream &out=mfem::out, int width=8) { Vector r(data.data(), data.size()); r.Print(out, width);}
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
      size = GetNP()*FieldVDims[f];
      fields[f] = Vector(data.data() + offset, size);
      offset += fields[f].Size();
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

      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0) // If processing coord comps
            {
               dat = p.GetCoords()[c];
            }
            else if (f - 1 < NumScalars) // Else if processing scalars
            {
               dat = p.GetScalar(f - 1);
            }
            else // Else processing vector comps
            {
               dat = p.GetVector(f - 1 - NumScalars)[c];
            }
            data.insert(data.begin() + offset, dat);
            offset += old_np + 1; // 1 to account for added data each loop iteration
         }
      }
   }
   else // byVDIM
   {
      const real_t* dat;
      for (int f = 0; f < TotalFields; f++)
      {
         if (f == 0)
         {
            dat = p.GetCoords().GetData();
         }
         else if (f - 1 < NumScalars)
         {
            dat = &p.GetScalar(f-1);
         }
         else
         {
            dat = p.GetVector(f-1-NumScalars).GetData();
         }
         data.insert(data.begin() + old_np*(ExclScanFieldVDims[f] + FieldVDims[f]) + ExclScanFieldVDims[f], dat, dat + FieldVDims[f]);
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
      for (int i = sorted_list[0]; i < data.size(); i++)
      {
         if (i % GetNP() == sorted_list[rm_count % sorted_list.Size()])
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
      int rm_count = 0;

      int f = 0;
      for (int i = sorted_list[0]*FieldVDims[0]; i < data.size();  i++)
      {
         if (i == FieldVDims[f]*GetNP())
         {
            f++;
         }
         if ( (i / FieldVDims[f]) % ( ExclScanFieldVDims[f]*GetNP() ) == sorted_list[(rm_count / FieldVDims[f]) % sorted_list.Size()])
         {
            rm_count++;
         }
         else
         {
            data[i-rm_count] = data[i];
         }
      }
   }

   // Resize / remove tail
   int num_new = old_np - list.Size();
   data.resize(num_new*TotalComps);

   SyncVectors();
}


template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
Particle<SpaceDim, NumScalars, VectorVDims...> ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::GetParticleRef(int i)
{
   static_assert(VOrdering == Ordering::byVDIM, "GetParticleRef is only available when ordering is byVDIM.");

   real_t *coords = &data[i*SpaceDim];

   real_t *scalars[NumScalars];
   for (int s = 0; s < NumScalars; s++) scalars[s] = &data[(s+SpaceDim)*GetNP() + i];

   real_t *vectors[sizeof...(VectorVDims)];
   for (int v = 0; v < sizeof...(VectorVDims); v++) vectors[v] = &data[ExclScanFieldVDims[v+1+NumScalars]*GetNP() + i*VDims[v]];

   return Particle<SpaceDim, NumScalars, VectorVDims...>(coords, scalars, vectors);
   
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
Particle<SpaceDim, NumScalars, VectorVDims...> ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::GetParticleData(int i) const
{
   if constexpr(VOrdering == Ordering::byNODES)
   {
      real_t *dat;
      Particle<SpaceDim, NumScalars, VectorVDims...> p;
      for (int f = 0; f < TotalFields; f++)
      {
         for (int c = 0; c < FieldVDims[f]; c++)
         {
            if (f == 0)
            {
               dat = &p.GetCoords()[c];
            }
            else if (f-1 < NumScalars)
            {
               dat = &p.GetScalar(f-1);
            }
            else
            {
               dat = &p.GetVector(f-1-NumScalars)[c];
            }
            *dat = data[i+(c+ExclScanFieldVDims[f])*GetNP()];
         }
      }
      return Particle(p);
   }
   else // byVDIM
   {
      return Particle<SpaceDim, NumScalars, VectorVDims...>(GetParticleRef(i));
   }
   
}

template<int SpaceDim, int NumScalars, int... VectorVDims, Ordering::Type VOrdering>
void ParticleSet<Particle<SpaceDim,NumScalars,VectorVDims...>, VOrdering>::SetParticle(int i, const Particle<SpaceDim, NumScalars, VectorVDims...> &p)
{
   // TODO
   /*
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
   */
}

} // namespace mfem


#endif // MFEM_PARTICLE_SET