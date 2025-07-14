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


#ifndef MFEM_PARTICLEDATA
#define MFEM_PARTICLEDATA

#include "../linalg/linalg.hpp"
#include "fespace.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

// Forward declaration
class ParticleSpace;


// TODO: We want the data to be more-easily exposed as more natural types.... (Array<int> and Vector)
//       I honestly do like ParticleFunction... Should I also have just a ParticleArray?
//       (For both of them, we can easily set memory....)
// TODO: Should we have operator[] in the base or in the derived on the more natural types? Maybe both?
//       (Thinking about Capacity() v. Size())

// Arbitrary data types may be associated with particles
// Only int and real_t supported right now... Can support more.
template<typename T>
class ParticleData
{
   friend class ParticleSpace;
   
protected:
   const int vdim;
   const Ordering::Type ordering;
   
   int np; // num particles = data.Capacity()/vdim --- here so we don't need to recompute each time
            // Future: Can use to track active v. inactive particles....?

   // All particle data is now stored entirely in Memory<T>
   // (Much more GPU-ready)
   // TODO: IF we can create a "DynamicMemory", that would be perfect.
   // TODO Maybe we also just make this an Array<T> ??? 
   Memory<T> data;

   // Only allow ParticleSpace to resize data / update data
   // TODO: Can make these faster if a "Reserve" is provided....
   //          See Vector....
   // Now it's just these sole two fxns below that we need to optimize....
   void AddParticles(int num_new);
   void RemoveParticles(const Array<int> &indices);

   // Maybe we have a "SyncWrapper" function that's abstract + called at end of AddParticles + RemoveParticles
   // to sync a wrapper? (Vector for ParticleFunction, Array for ParticleArray???
   void SyncWrapper() = 0;

   // No public ctor -- only constructable through a ParticleSpace or derived
   // Ordering irrelevant for vdim = 1
   ParticleData(int num_particles, int vdim_=1, Ordering::Type ordering_=Ordering::byNODES)
   : vdim(vdim_), ordering(ordering_), np(num_particles), data(0) { AddParticles(np); SyncWrapper(); }

   T& GetParticleData(int i, int comp=0);

   const T& GetParticleData(int i, int comp=0) const;

   void GetParticleData(int i, Memory<T> &pdata) const;

   void SetParticleData(int i, const T &pdata, int comp=0);

   void SetParticleData(int i, const Memory<T> &pdata);

   // Set multiple particles' data, given particle indices
   // Ordering must match that of the ParticleSpace
   void SetParticleData(const Array<int> &indices, const Memory<T> &pdatas);

   T& operator[](int idx) { return data[idx]; }

   const T& operator[](int idx) const { return data[idx]; }

public:
   // No public methods
};



class ParticleArray : public ParticleData<int>
{

};

class ParticleFunction : public ParticleData<real_t>
{
   friend class ParticleSpace;
   
private:
   const ParticleSpace &pspace;

   Vector v_data;

protected:

   ParticleFunction(const ParticleSpace &pspace, int vdim_=1);

   void SyncWrapper() { v_data.NewMemoryAndSize(data, np, false); }

public:
   
   // Interpolate a GridFunction onto the particles' locations
   // Automatically checks if Mesh was registered with FindPointsGSLIB in ParticleSpace
   // (and uses it if so! Saves a Setup and a FindPoints step...)
   void Interpolate(GridFunction &gf);

   // Project a coefficient
   // There MUST be a mesh registered to the ParticleSpace for this to work...
   void ProjectCoefficient(Coefficient &coeff, int mesh_idx=0);

   void ProjectCoefficient(VectorCoefficient &vcoeff, int mesh_idx=0);

   void GetParticleData(int i, Vector &pdata) const
   { ParticleData<real_t>::GetParticleData(i, pdata.GetMemory()); }

   void SetParticleData(int i, const Vector &pdata)
   { ParticleData<real_t>::SetParticleData(i, pdata.GetMemory()); }

   // Set multiple particles' data, given particle indices
   // Ordering must match that of the ParticleSpace
   void SetParticleData(const Array<int> &indices, const Vector &pdatas)
   { ParticleData<real_t>::SetParticleData(indices, pdatas.GetMemory()); }

   const Vector& GetVector() const { return v_data; }


};



} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_PARTICLEDATA