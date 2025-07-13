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

#include "particlespace.hpp"
#include "../linalg/linalg.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

template<typename T>
class ParticleData
{
protected:
   const ParticleSpace &pspace;
   const int vdim;
   int reg_idx;

   Memory<T> data;

public:
   ParticleData(ParticleSpace &pspace_, int vdim_=1, bool register_data=true);

   T& GetParticleData(int i, int comp=0);

   // For byVDIM, pdata is an alias to the actual daata
   // For byNODES, pdata is a copy of the actual data
   void GetParticleData(int i, Memory<T> &pdata);

   void SetParticleData(int i, const T &ParticleData, int comp=0);

   void SetParticleData(int i, const Memory<T> &pdata);

   // Ordering must match that of the ParticleSpace
   void SetData(const Array<int> &indices, const Memory<T> &pdatas);

   void RemoveData(const Array<int> &indices);

   ~ParticleData();

};



} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_PARTICLEDATA