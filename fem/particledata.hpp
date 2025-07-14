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


template<typename T>
class ParticleData
{
   friend class ParticleSpace;
   
protected:
   const int vdim;
   const Ordering::Type ordering;
   
   int np; // (number of particles * vdim) < data.Capacity() in general

   // All particle data is now stored entirely in Memory<T>
   Memory<T> data;

   // Resize properly to hold num_new added particles
   void AddParticles(int num_new);

   // Resize properly to remove particles at indices
   void RemoveParticles(const Array<int> &indices);

   // Re-allocates so np*vdim = Capacity()
   void ShrinkToFit();

   // "SyncWrapper" function that's abstract + called at end of AddParticles + RemoveParticles
   // --- To sync a more user-friendly wrapper (Vector for ParticleFunction, Array<int> for ParticleArray<int>)
   virtual void SyncWrapper() = 0;

   // No public ctor -- only constructable through a ParticleSpace or derived
   ParticleData(int reserve, int vdim_=1, Ordering::Type ordering_=Ordering::byNODES)
   : vdim(vdim_), ordering(ordering_), np(0), data(reserve*vdim) {}

   T& GetParticleData(int i, int comp=0);

   const T& GetParticleData(int i, int comp=0) const;

   void GetParticleData(int i, Memory<T> &pdata) const;

   void SetParticleData(int i, const T &pdata, int comp=0);

   // Set multiple particles' data, given particle indices
   // Ordering must match that of the ParticleSpace
   void SetParticleData(const Array<int> &indices, const Memory<T> &pdatas);

   void SetParticleData(int i, const Memory<T> &pdata)
   { SetParticleData(Array<int>({i}), pdata); }

public:

   int Size() const { return np*vdim; }

   int Capacity() const { return data.Capacity(); }

};


template<typename T>
class ParticleArray : public ParticleData<T>
{
   friend class ParticleSpace;
protected:
   Array<T> a_data;

   void SyncWrapper() override { a_data.MakeRef(this->data, this->np, this->data.GetMemoryType(), false); }

   ParticleArray(int num_particles)
   : ParticleData<T>(num_particles) { ParticleData<T>::AddParticles(num_particles); };
   // TODO: For now, capacity == num_particles...

public:

   void SetParticleData(const Array<int> &indices, const Array<T> &pdatas)
   { ParticleData<T>::SetParticleData(indices, pdatas.GetMemory()); }

   // Don't need specialized getters/setters when vdim == 1
   T& operator[](int idx) { return a_data[idx]; }

   const T& operator[](int idx) const { return a_data[idx]; }

   const Array<int>& GetArray() const { return a_data; }

};


class ParticleFunction : public ParticleData<real_t>
{
   friend class ParticleSpace;
   
protected:
   const ParticleSpace &pspace;

   Vector v_data;

   ParticleFunction(const ParticleSpace &pspace, int vdim_=1);

   void SyncWrapper() { v_data.NewMemoryAndSize(this->data, this->np, false); }

public:

   real_t& GetParticleData(int i, int comp=0)
   { return ParticleData<real_t>::GetParticleData(i, comp); }

   const real_t& GetParticleData(int i, int comp=0) const
   { return ParticleData<real_t>::GetParticleData(i, comp); }

   void GetParticleData(int i, Vector &pdata) const
   { ParticleData<real_t>::GetParticleData(i, pdata.GetMemory()); }

   void SetParticleData(int i, const real_t &pdata, int comp=0)
   { return ParticleData<real_t>::SetParticleData(i, pdata, comp); }

   void SetParticleData(int i, const Vector &pdata)
   { ParticleData<real_t>::SetParticleData(i, pdata.GetMemory()); }

   // Set multiple particles' data, given particle indices
   // Ordering must match that of the ParticleSpace
   void SetParticleData(const Array<int> &indices, const Vector &pdatas)
   { ParticleData<real_t>::SetParticleData(indices, pdatas.GetMemory()); }

   real_t& operator[](int idx) { return v_data[idx]; }

   const real_t& operator[](int idx) const { return v_data[idx]; }

   const Vector& GetVector() const { return v_data; }


   // Below functions are all TODO still...

   // Interpolate a GridFunction onto the particles' locations
   // Automatically checks if Mesh was registered with FindPointsGSLIB in ParticleSpace
   // (and uses it if so! Saves a Setup and a FindPoints step...)
   void Interpolate(GridFunction &gf);

   // Project a coefficient
   // There MUST be a mesh registered to the ParticleSpace for this to work...
   void ProjectCoefficient(Coefficient &coeff, int mesh_idx=0);

   void ProjectCoefficient(VectorCoefficient &vcoeff, int mesh_idx=0);

};



} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_PARTICLEDATA