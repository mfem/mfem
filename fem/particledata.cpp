
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
#include "particledata.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

template<typename T>
ParticleData<T>::ParticleData(ParticleSpace &pspace_, int vdim_=1,
                              bool register_data=true)
   : pspace(pspace_),
     vdim(vdim_),
     reg_idx(-1),
     data(vdim*pspace.GetNP())
{

   if (register_data)
   {
      reg_idx = pspace.Register(*this);
   }
}

template<typename T>
T& ParticleData<T>::GetParticleData(int i, int comp)
{
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      return data[i + comp*pspace.GetNP()];
   }
   else
   {
      return data[comp + i*vdim];
   }
}

template<typename T>
void ParticleData<T>::GetParticleData(int i, Memory<T> &pdata)
{
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      pdata.New(vdim);
      for (int c = 0; c < vdim; c++)
      {
         pdata[i] = data[i + c*pspace.GetNP()];
      }
   }
   else
   {
      pdata.MakeAlias(data, i*vdim, vdim);
   }
}

template<typename T>
void ParticleData<T>::SetParticleData(int i, const T &pdata, int comp)
{
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      data[i + comp*pspace.GetNP()] = pdata;
   }
   else
   {
      data[comp + i*vdim] = pdata;
   }
}

template<typename T>
void ParticleData<T>::SetParticleData(int i, const Memory<T> &pdata)
{
   MFEM_ASSERT(pdata.Capacity() == vdim,
               "Input Memory<T> has capacity " + std::to_string(pdata.Capacity()),
               ", not vdim " + std::to_string(vdim));

   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      for (int c = 0; c < vdim; c++)
      {
         data[i + c*pspace.GetNP()] = pdata[c];
      }
   }
   else
   {
      for (int c = 0; c < vdim; c++)
      {
         data[comp + i*vdim] = pdata[c];
      }
   }
}

template<typename T>
void ParticleData<T>::SetData(const Array<int> &indices,
                              const Memory<T> &pdatas)
{
   MFEM_ASSERT(pdatas.Capacity()/vdim == indices.Size(),
               "Indices size incompatible with input pdatas");


}

template<typename T>
ParticleData<T>::~ParticleData()
{
   if (reg_idx >= 0)
   {
      pspace.DeregisterParticleData(reg_idx);
   }
}


// Only real_t and int supported currently
template class ParticleData<real_t>;
template class ParticleData<int>;

} // namespace mfem

#endif // MFEM_USE_GSLIB