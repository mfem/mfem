
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
void ParticleData<T>::AddNewParticleData(int num_new)
{
   // TODO: Definitely some mistakes with memorytype + usedevice stuff...
   Memory<T> old = data;
   data.Delete();
   data.New(num_new*vdim+old.Capacity(), old.GetMemoryType());
   data.UseDevice(old.UseDevice());

   // Fill in old data entries. Leave new data as default-initialization
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      int total_old = old.Capacity()/vdim;
      int total_new = data.Capacity()/vdim;
      for (int i = 0; i < old.Capacity(); i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[i+c*total_new] = old[i+c*total_old];
         }
      }
   }
   else
   {
      for (int i = 0; i < old.Capacity(); i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[c+i*vdim] = old[c+i*vdim];
         }
      }
   }
   old.Delete();
}

template<typename T>
void ParticleData<T>::RemoveData(const Array<int> &indices)
{
   // TODO: Definitely some mistakes with memorytype + usedevice stuff...
   Memory<T> old = data;
   data.Delete();
   data.New(old.Capacity(), old.GetMemoryType());
   data.UseDevice(old.UseDevice());


   // Copy non-removed data over

   // TODO!
}

template<typename T>
ParticleData<T>::ParticleData(ParticleSpace &pspace_, int vdim_, bool register_to_pspace)
: pspace(pspace_),
  vdim(vdim_),
  data(vdim*pspace.GetNP())
{
   pspace.RegisterParticleData(*this);
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
      if (pdata.Capacity() != vdim)
      {
         // TODO: Definitely some mistakes with memorytype + usedevice stuff...
         const MemoryType mt = pdata.GetMemoryType();
         const bool use_dev = pdata.UseDevice();
         pdata.Delete();
         pdata.New(vdim, mt);
         pdata.UseDevice(use_dev);
      }
      for (int c = 0; c < vdim; c++)
      {
         pdata[i] = data[i + c*pspace.GetNP()];
      }
   }
   else
   {
      pdata.Delete();
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
   // MFEM_ASSERT(pdata.Capacity() == vdim,
   //             "Input Memory<T> has capacity " + std::to_string(pdata.Capacity()),
   //             ", not vdim " + std::to_string(vdim));

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
void ParticleData<T>::SetParticleData(const Array<int> &indices,
                              const Memory<T> &pdatas)
{
   // MFEM_ASSERT(pdatas.Capacity()/vdim == indices.Size(),
   //             "Indices size incompatible with input pdatas");

   
   if (pspace.GetOrdering() == Ordering::byNODES)
   {
      int num_update = pdatas.Capacity()/vdim;

      for (int i = 0; i < indices.Size(); i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[indices[i]+c*pspace.GetNP()] = pdata[i+c*num_update];
         }
      }
   }
   else
   {
      for (int i = 0; i < indices.Size(); i++)
      {
         for (int c = 0; c < vdim; c++)
         {
            data[c+indices[i]*vdim] = pdata[c+i*vdim];
         }
      }
   }
}

template<typename T>
ParticleData<T>::~ParticleData()
{
   pspace.
}

// Only real_t and int supported currently
template class ParticleData<real_t>;
template class ParticleData<int>;

} // namespace mfem

#endif // MFEM_USE_GSLIB