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

#ifdef MFEM_USE_GSLIB

namespace mfem
{

// Forward declaration
class ParticleSpace;


// Arbitrary data types may be associated with particles
// Only int and real_t supported right now... Can support more.
template<typename T>
class ParticleData
{
   friend class ParticleSpace;
   
protected:
   const int vdim;
   ParticleSpace *pspace;
   
   // All particle data is now stored entirely in Memory<T>
   // (Much more GPU-ready)
   // TODO: IF we can create a "DynamicMemory", that would be perfect.
   Memory<T> data;

   // Only allow ParticleSpace to resize data / update data
   // TODO: Can make these faster if a "Reserve" is provided....
   //          See Vector....
   // Now it's just these sole two fxns below that we need to optimize....
   void AddNewParticleData(int num_new);
   void RemoveData(const Array<int> &indices);

public:
   ParticleData(ParticleSpace &pspace_, int vdim_=1, bool register_to_pspace=true);

   T& GetParticleData(int i, int comp=0);

   // For byVDIM, pdata is an alias to the actual daata
   // For byNODES, pdata is a copy of the actual data
   void GetParticleData(int i, Memory<T> &pdata);

   void SetParticleData(int i, const T &pdata, int comp=0);

   void SetParticleData(int i, const Memory<T> &pdata);

   // Set many particle data, given particle indices
   // Ordering must match that of the ParticleSpace
   void SetParticleData(const Array<int> &indices, const Memory<T> &pdatas);

   T& operator[](int idx) { return data[idx]; }

   const T& operator[](int idx) { return data[idx]; }

   ~ParticleData();

};

// User-friendly ParticleData<real_t>, with Interpolate feature
class ParticleFunction : public ParticleData<real_t>
{
private:
protected:
public:

}



} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_PARTICLEDATA